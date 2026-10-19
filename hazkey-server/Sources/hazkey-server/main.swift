import Foundation

do {
    NSLog("Starting hazkey-server...")
    let server = HazkeyServer()

    try server.start()
} catch {
    NSLog("Failed to start server: \(error)")
    exit(1)
}
