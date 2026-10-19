// FIXME: Lock
final public class PublicAzkkcApi: @unchecked Sendable {
    public static let shared = PublicAzkkcApi()
    private var NumGpuLayers: Int32

    private init() {
        NumGpuLayers = 13
    }

    public func setGpuLayers(_ layer: Int32) {
        NumGpuLayers = layer
    }

    public func getGpuLayers() -> Int32 {
        return NumGpuLayers
    }
}
