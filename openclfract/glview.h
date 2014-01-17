//
//  Author  : Matti Määttä
//  Summary : Qt OpenGL window widget
//

#ifndef GLVIEW_H
#define GLVIEW_H

#include <QtOpenGL>

#include "openclpbo.h"

class GLView : public QGLWidget
{
    Q_OBJECT

public:
    GLView(QWidget* parent = nullptr);

protected:
    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();

    void keyPressEvent(QKeyEvent* event);
    void wheelEvent(QWheelEvent* event);
    void mouseMoveEvent(QMouseEvent* event);

private:
    OpenCLPBO pbo_;
    unsigned long frames_;
    OpenCLPBO::Color color_;
    std::string error_;
    int samples_;

    // Fractal viewport
    double minRe_;
    double maxRe_;
    double minIm_;
    double maxIm_;

    QPoint lastPos_;

    void drawInfo(int elapsed, int kernel);
    void drawError(const std::string& what);
    void setView();
};

#endif // GLVIEW_H