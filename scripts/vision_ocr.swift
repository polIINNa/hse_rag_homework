// OCR сканов на macOS. Результат содержит координаты строк, а не только текст.
import Foundation
import Vision
import PDFKit
import AppKit

guard CommandLine.arguments.count == 4 else {
    fputs("Usage: vision_ocr PDF PAGE OUTPUT_JSON\n", stderr)
    exit(2)
}
let input = URL(fileURLWithPath: CommandLine.arguments[1])
let number = Int(CommandLine.arguments[2])! - 1
guard let doc = PDFDocument(url: input), let page = doc.page(at: number) else {
    fputs("Cannot open PDF page\n", stderr)
    exit(1)
}
let bounds = page.bounds(for: .mediaBox)
let scale: CGFloat = 3
let width = Int(bounds.width * scale)
let height = Int(bounds.height * scale)
let color = CGColorSpaceCreateDeviceRGB()
let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: color,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
ctx.setFillColor(CGColor(gray: 1, alpha: 1))
ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
ctx.scaleBy(x: scale, y: scale)
page.draw(with: .mediaBox, to: ctx)
let image = ctx.makeImage()!
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["ru-RU", "en-US"]
request.usesLanguageCorrection = false
try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
let rows = (request.results ?? []).compactMap { observation -> [String: Any]? in
    guard let candidate = observation.topCandidates(1).first else { return nil }
    let box = observation.boundingBox
    return ["text": candidate.string, "confidence": candidate.confidence,
            "x0": box.minX * bounds.width, "x1": box.maxX * bounds.width,
            "y0": (1 - box.maxY) * bounds.height, "y1": (1 - box.minY) * bounds.height]
}
let data = try JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys])
try data.write(to: URL(fileURLWithPath: CommandLine.arguments[3]), options: .atomic)

