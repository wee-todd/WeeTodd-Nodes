import SwiftUI
import StudioCore

struct RenderStatsView: View {
  let stats: RenderStats?
  var body: some View {
    if let stats {
      VStack(alignment: .leading, spacing: 3) {
        if let elapsed = stats.elapsedSeconds {
          Text("Render time · \(RenderStats.duration(elapsed))")
        }
        if let sampling = stats.samplingSeconds {
          Text("\(stats.samplingScope ?? "Sampling") · \(RenderStats.duration(sampling))")
            .help("LTX pre-decode timing includes prompt encoding, model loading, sampling and latent upscaling. H3 reports transformer sampling time.")
        }
        if let peak = stats.processPeakBytes {
          Text("Process peak · \(peak / 1_000_000_000, specifier: "%.2f") GB")
            .help("Peak resident memory of this renderer process; not total system memory.")
        }
        if let peak = stats.mlxPeakBytes {
          Text("\(stats.mlxPeakScope ?? "MLX") peak · \(peak / 1_000_000_000, specifier: "%.2f") GB")
            .help("MLX allocations measured within this scope. This is not total process or system RAM.")
        }
      }.font(.caption.monospacedDigit()).foregroundStyle(.secondary)
    }
  }
}
