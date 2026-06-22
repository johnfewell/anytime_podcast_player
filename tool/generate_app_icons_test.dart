// One-shot asset generator for the "Ambient" rebrand. Run with:
//
//   flutter test tool/generate_app_icons_test.dart
//
// It paints the minimal Ambient ripple mark (a centred dot inside two
// concentric rings) onto a few canvases and writes the PNGs that
// flutter_launcher_icons and the native launch screens consume. This is a
// build tool, not a real test — it lives under tool/ so CI's `test/` run
// ignores it. Re-run it whenever the mark or brand colours change.

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

const Color _blue = Color(0xff0075de); // Notion Blue — the one action colour
const Color _white = Color(0xffffffff);

/// Paints the Ambient ripple: a filled centre dot wrapped by two concentric
/// rings. [coverage] scales the whole mark within the square so we can keep it
/// inside the adaptive-icon safe zone or give it more presence on a filled tile.
void _drawMark(Canvas canvas, double size, Color color, double coverage) {
  final centre = Offset(size / 2, size / 2);
  double r(double fraction) => size * fraction * coverage;

  final fill = Paint()
    ..color = color
    ..isAntiAlias = true;
  canvas.drawCircle(centre, r(0.075), fill);

  final ring = Paint()
    ..color = color
    ..style = PaintingStyle.stroke
    ..strokeCap = StrokeCap.round
    ..isAntiAlias = true
    ..strokeWidth = size * 0.058 * coverage;
  canvas.drawCircle(centre, r(0.17), ring);
  canvas.drawCircle(centre, r(0.27), ring);
}

Future<void> _emit(
  String path,
  int size,
  void Function(Canvas canvas, double size) paint,
) async {
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder);
  paint(canvas, size.toDouble());
  final picture = recorder.endRecording();
  final image = await picture.toImage(size, size);
  final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
  picture.dispose();
  image.dispose();

  final file = File(path);
  file.parent.createSync(recursive: true);
  file.writeAsBytesSync(bytes!.buffer.asUint8List());
  // ignore: avoid_print
  print('wrote $path (${size}x$size)');
}

void main() {
  testWidgets('generate Ambient app icons + launch marks', (tester) async {
    await tester.runAsync(() async {
      // Full launcher tile: white ripple on a solid Notion-blue field.
      await _emit('assets/images/ambient-icon.png', 1024, (canvas, s) {
        canvas.drawRect(Rect.fromLTWH(0, 0, s, s), Paint()..color = _blue);
        _drawMark(canvas, s, _white, 1.0);
      });

      // Adaptive foreground: transparent, mark pulled in to the safe zone.
      await _emit('assets/images/ambient-icon-fg.png', 1024, (canvas, s) {
        _drawMark(canvas, s, _white, 0.8);
      });

      // Launch-screen mark: blue ripple on transparent, sits on warm paper.
      await _emit('assets/images/ambient-mark.png', 512, (canvas, s) {
        _drawMark(canvas, s, _blue, 0.8);
      });

      // Android launch drawable (no-dpi so it isn't density-scaled).
      await _emit(
        'android/app/src/main/res/drawable-nodpi/splash_mark.png',
        288,
        (canvas, s) => _drawMark(canvas, s, _blue, 0.8),
      );

      // iOS launch image (@1x/@2x/@3x slots all point at 512px files).
      for (final name in const [
        'anytime-logo-s.png',
        'anytime-logo-s 1.png',
        'anytime-logo-s 2.png',
      ]) {
        await _emit(
          'ios/Runner/Assets.xcassets/LaunchImage.imageset/$name',
          512,
          (canvas, s) => _drawMark(canvas, s, _blue, 0.8),
        );
      }
    });
  });
}
