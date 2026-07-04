use anyhow::{Context, Result, anyhow};
use clap::Parser;
use serde::Serialize;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Parser, Debug)]
#[command(name = "meetily-cli", about = "Transcribe audio files using Whisper")]
struct Args {
    /// Audio file to transcribe (MP4, M4A, WAV, MP3, FLAC, OGG, MKV, WebM, WMA)
    #[arg(long, short)]
    file: PathBuf,

    /// Whisper model name (without .bin extension)
    #[arg(long, short, default_value = "large-v3-turbo-q5_0")]
    model: String,

    /// Directory containing downloaded .bin model files
    #[arg(long, default_value = "~/.meetily/models")]
    models_dir: String,

    /// Language code ("en", "es", etc.) or "auto" for detection
    #[arg(long, short, default_value = "auto")]
    language: String,

    /// Whisper thread count (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: i32,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    Progress { stage: String, percent: u32, message: String },
    Segment { start: f64, end: f64, text: String },
    Done { segments_count: usize, duration_seconds: f64 },
    Error { message: String },
}

fn emit(event: &Event) {
    let json = serde_json::to_string(event).unwrap_or_default();
    println!("{}", json);
    io::stdout().flush().ok();
}

fn progress(stage: &str, percent: u32, message: impl Into<String>) {
    emit(&Event::Progress { stage: stage.to_string(), percent, message: message.into() });
}

fn resolve_models_dir(raw: &str) -> Result<PathBuf> {
    if raw.starts_with('~') {
        let home = dirs::home_dir().ok_or_else(|| anyhow!("Cannot find home directory"))?;
        Ok(home.join(raw.trim_start_matches("~/")))
    } else {
        Ok(PathBuf::from(raw))
    }
}

fn main() {
    if let Err(e) = run() {
        emit(&Event::Error { message: format!("{:#}", e) });
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    let models_dir = resolve_models_dir(&args.models_dir)?;
    let model_path = models_dir.join(if args.model.starts_with("ggml-") { format!("{}.bin", args.model) } else { format!("ggml-{}.bin", args.model) });

    if !model_path.exists() {
        return Err(anyhow!(
            "Model not found: {}. Download it via the Meetily app or place the .bin file there.",
            model_path.display()
        ));
    }
    if !args.file.exists() {
        return Err(anyhow!("Audio file not found: {}", args.file.display()));
    }

    // Step 1: Prepare (ffmpeg for MKV/WebM/WMA)
    progress("decode", 5, format!("Preparing {}", args.file.display()));
    let (audio_path, needs_cleanup) = prepare_audio(&args.file)?;

    // Step 2: Decode to f32 samples
    progress("decode", 10, "Decoding audio");
    let result = decode_audio(&audio_path);
    if needs_cleanup {
        let _ = std::fs::remove_file(&audio_path);
    }
    let (samples, sample_rate, duration_seconds) = result?;
    progress("decode", 20, format!(
        "Decoded {:.1}s of audio at {}Hz",
        duration_seconds, sample_rate
    ));

    // Step 3: Resample to 16kHz
    let samples_16k = if sample_rate != 16000 {
        progress("resample", 25, format!("Resampling {}Hz → 16000Hz", sample_rate));
        let out = resample_to_16k(&samples, sample_rate)?;
        progress("resample", 35, "Resample complete");
        out
    } else {
        progress("resample", 35, "Already at 16kHz");
        samples
    };

    // Step 4: Load model
    progress("load", 40, format!("Loading {}", args.model));
    std::env::set_var("GGML_METAL_LOG_LEVEL", "1");
    std::env::set_var("WHISPER_LOG_LEVEL", "1");

    let ctx = WhisperContext::new_with_params(
        model_path.to_str().ok_or_else(|| anyhow!("Invalid model path"))?,
        WhisperContextParameters::default(),
    )
    .context("Failed to load Whisper model")?;

    progress("load", 55, "Model loaded");

    // Step 5: Transcribe
    progress("transcribe", 60, "Transcribing");
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    let threads = if args.threads > 0 {
        args.threads
    } else {
        (std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4) / 2).max(1)
    };
    params.set_n_threads(threads);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_print_special(false);
    params.set_translate(false);
    params.set_no_context(true);

    if args.language != "auto" {
        params.set_language(Some(&args.language));
    }

    let mut state = ctx.create_state().context("Failed to create Whisper state")?;
    state.full(params, &samples_16k).context("Transcription failed")?;

    progress("transcribe", 90, "Extracting segments");

    let n = state.full_n_segments().context("Failed to get segment count")?;
    let mut segments_count = 0usize;

    for i in 0..n {
        let text = match state.full_get_segment_text(i) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let text = text.trim().to_string();
        if text.is_empty() { continue; }

        let t0 = state.full_get_segment_t0(i).unwrap_or(0);
        let t1 = state.full_get_segment_t1(i).unwrap_or(0);

        emit(&Event::Segment {
            start: t0 as f64 / 100.0,
            end: t1 as f64 / 100.0,
            text,
        });
        segments_count += 1;
    }

    emit(&Event::Done { segments_count, duration_seconds });
    Ok(())
}

/// Convert MKV/WebM/WMA to a temp WAV via ffmpeg. Returns (path, needs_cleanup).
fn prepare_audio(path: &Path) -> Result<(PathBuf, bool)> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if ["mkv", "webm", "wma"].contains(&ext.as_str()) {
        progress("decode", 3, format!("Converting {} via ffmpeg", ext));
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let tmp = std::env::temp_dir().join(format!("meetily-{}.wav", secs));
        let status = Command::new("ffmpeg")
            .args(["-y", "-i", path.to_str().unwrap_or(""),
                   "-ar", "16000", "-ac", "1", "-f", "wav",
                   tmp.to_str().unwrap_or("")])
            .stderr(std::process::Stdio::null())
            .status()
            .context("ffmpeg not found — install ffmpeg for MKV/WebM/WMA support")?;
        if !status.success() {
            return Err(anyhow!("ffmpeg conversion failed for {}", path.display()));
        }
        return Ok((tmp, true));
    }
    Ok((path.to_path_buf(), false))
}

/// Decode audio file to mono f32 samples at native sample rate.
fn decode_audio(path: &Path) -> Result<(Vec<f32>, u32, f64)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("Unsupported or corrupt audio format")?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No audio track found"))?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let n_channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(
            &format.tracks().iter().find(|t| t.id == track_id).unwrap().codec_params,
            &DecoderOptions::default(),
        )
        .context("Failed to create audio decoder")?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(symphonia::core::errors::Error::ResetRequired) => { decoder.reset(); continue; }
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id { continue; }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                buf.copy_interleaved_ref(decoded);
                all_samples.extend_from_slice(buf.samples());
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        }
    }

    let mono: Vec<f32> = if n_channels > 1 {
        all_samples.chunks(n_channels)
            .map(|ch| ch.iter().sum::<f32>() / n_channels as f32)
            .collect()
    } else {
        all_samples
    };

    let duration = mono.len() as f64 / sample_rate as f64;
    Ok((mono, sample_rate, duration))
}

/// Resample mono f32 from `from_rate` Hz to 16000 Hz.
fn resample_to_16k(samples: &[f32], from_rate: u32) -> Result<Vec<f32>> {
    use rubato::{FftFixedIn, Resampler};

    let chunk_size = 2048usize;
    let mut resampler = FftFixedIn::<f32>::new(from_rate as usize, 16000, chunk_size, 2, 1)
        .context("Failed to create resampler")?;

    let input_size = resampler.input_frames_next();
    let mut output: Vec<f32> = Vec::with_capacity(
        (samples.len() as f64 * 16000.0 / from_rate as f64) as usize + 4096,
    );

    let mut pos = 0usize;
    while pos + input_size <= samples.len() {
        let chunk = vec![samples[pos..pos + input_size].to_vec()];
        let out = resampler.process(&chunk, None)?;
        output.extend_from_slice(&out[0]);
        pos += input_size;
    }

    if pos < samples.len() {
        let mut tail = samples[pos..].to_vec();
        let tail_len = tail.len();
        tail.resize(input_size, 0.0);
        let out = resampler.process(&[tail], None)?;
        let keep = ((tail_len as f64 * 16000.0 / from_rate as f64) as usize).min(out[0].len());
        output.extend_from_slice(&out[0][..keep]);
    }

    for s in &mut output {
        *s = s.clamp(-1.0, 1.0);
    }

    Ok(output)
}
