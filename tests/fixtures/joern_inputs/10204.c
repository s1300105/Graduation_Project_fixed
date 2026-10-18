void QPaintEngineEx::drawPixmapFragments(const QPainter::PixmapFragment *fragments, int fragmentCount,
                                         const QPixmap &pixmap, QPainter::PixmapFragmentHints /*hints*/)
{
    if (pixmap.isNull())
        return;

    qreal oldOpacity = state()->opacity;
    QTransform oldTransform = state()->matrix;

    for (int i = 0; i < fragmentCount; ++i) {
        QTransform transform = oldTransform;
        transform.translate(fragments[i].x, fragments[i].y);
        transform.rotate(fragments[i].rotation);
        state()->opacity = oldOpacity * fragments[i].opacity;
        state()->matrix = transform;
        opacityChanged();
        transformChanged();

        qreal w = fragments[i].scaleX * fragments[i].width;
        qreal h = fragments[i].scaleY * fragments[i].height;
        QRectF sourceRect(fragments[i].sourceLeft, fragments[i].sourceTop,
                          fragments[i].width, fragments[i].height);
        drawPixmap(QRectF(-0.5 * w, -0.5 * h, w, h), pixmap, sourceRect);
    }

    state()->opacity = oldOpacity;
    state()->matrix = oldTransform;
    opacityChanged();
    transformChanged();
}