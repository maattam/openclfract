#include "glvew.h"

#include <QElapsedTimer>
#include <vector>

GLView::GLView(QWidget* parent)
    : QGLWidget(parent), frames_(0), lastPos_(0, 0),
    color_(OpenCLPBO::Poly), samples_(0)
{
    minRe_ = -1.0f;
    maxRe_ = 1.0f;
    minIm_ = -1.0f;
    maxIm_ = 1.0f;

    this->setMouseTracking(true);
}

void GLView::initializeGL()
{
    int vmaj = format().majorVersion();
    int vmin = format().minorVersion();
    if(vmaj < 2)
    {
        QMessageBox::warning(this, 
            tr("Wrong OpenGL version"), 
            tr("OpenGL version 2.0 or higher needed. You have %1.%2, so some functions may not work properly.").arg(vmaj).arg(vmin));
    }

    qDebug() << tr("OpenGL Version: %1.%2").arg(vmaj).arg(vmin);

    glClearColor(0, 0, 0, 0);
    glDisable(GL_DEPTH_TEST);
    glShadeModel(GL_FLAT);
    glDisable(GL_LIGHTING);

    try
    {
        pbo_.init("mandelbrot.cl", "mandelbrot");
    }

    catch(std::runtime_error& err)
    {
        QMessageBox::warning(this,
            tr("OpenCL Error"), tr("Failed to initialize OpenCLPBO: %1").arg(err.what()));

        return;
    }

    pbo_.precomputeColor(OpenCLPBO::Poly);
}

void GLView::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);

    // Reset the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Reset the modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    try
    {
        pbo_.resize(width * pow(2, samples_), height * pow(2, samples_));
    }

    catch(std::runtime_error& error)
    {
        error_ = error.what();
        return;
    }

    error_.clear();
}

void GLView::setView()
{
    if(width() > height())
    {
        double h = maxIm_ - minIm_;
        double fact =  (h - h / width() * height()) * 0.5;
        pbo_.setView(minRe_, maxRe_, minIm_ + fact, maxIm_ - fact);
    }

    else
    {
        double w = maxRe_ - minRe_;
        double fact = (w - w / height() * width()) * 0.5;
        pbo_.setView(minRe_ + fact, maxRe_ - fact, minIm_, maxIm_);
    }
}

void GLView::paintGL()
{
    int kernel = 0;
    QElapsedTimer timer;
    timer.start();

    setView();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Show pending error
    if(! error_.empty())
    {
        drawError(error_);
        return;
    }

    try
    {
        // Render fractal to texture memory
        kernel = pbo_.calculate();
    }

    catch(std::runtime_error& err)
    {
        drawError(err.what());
        return;
    }

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, pbo_.getTextureId());

    // Fill screen with a single quad
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f,1.0f);  glVertex3f(-1.0f,-1.0f,-1.0f);
        glTexCoord2f(0.0f,0.0f);  glVertex3f(-1.0f,1.0f,-1.0f);
        glTexCoord2f(1.0f,0.0f);  glVertex3f(1.0f,1.0f,-1.0f);
        glTexCoord2f(1.0f,1.0f);  glVertex3f(1.0f,-1.0f,-1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    drawInfo(timer.elapsed(), kernel);

    ++frames_;
}

void GLView::keyPressEvent(QKeyEvent* event)
{
    switch(event->key())
    {
    case Qt::Key_Escape: close(); break;

    case Qt::Key_Plus:
        {
            pbo_.max_iterations(pbo_.max_iterations() + 100);
            updateGL();
            break;
        }

    case Qt::Key_Minus:
        {
            pbo_.max_iterations(pbo_.max_iterations() - 100);
            updateGL();
            break;
        }

    case Qt::Key_C:
        {
            if(color_ == OpenCLPBO::Poly)
            {
                color_ = OpenCLPBO::Trig;
            }

            else
            {
                color_ = OpenCLPBO::Poly;
            }

            pbo_.precomputeColor(color_);
            updateGL();
            break;
        }

    case Qt::Key_A:
        {
            if(samples_ < 3)
            {
                samples_++;
                resizeGL(width(), height());
                updateGL();
            }

            break;
        }

    case Qt::Key_D:
        {
            if(samples_ > 0)
            {
                samples_--;
                resizeGL(width(), height());
                updateGL();
            }

            break;
        }

    case Qt::Key_Return:
        {
            if(event->modifiers() == Qt::AltModifier)
            {
                if(isFullScreen())
                {
                    showNormal();
                }

                else
                {
                    showFullScreen();
                }
            }
        }
    }

    event->accept();
}

void GLView::wheelEvent(QWheelEvent* event)
{
    int steps = event->delta() / 8 / 15;

    double scale = 0.05 * (maxRe_ - minRe_) * (double) steps;
    double kk = (minIm_ - maxIm_) / (maxRe_ - minRe_);

    minRe_ += scale;
    maxIm_ = kk * (minRe_ - maxRe_) + minIm_;

    maxRe_ -= scale;
    minIm_ = kk * (maxRe_ - minRe_) + maxIm_;

    updateGL();
    event->accept();
}

void GLView::mouseMoveEvent(QMouseEvent* event)
{
    int dx = event->x() - lastPos_.x();
    int dy = event->y() - lastPos_.y();

    if(event->buttons() & Qt::LeftButton)
    {
        double scaledX = (double) dx / width() * (maxRe_ - minRe_);
        double scaledY = (double) dy / height() * (maxIm_ - minIm_);

        maxRe_ -= scaledX;
        minRe_ -= scaledX;

        maxIm_ += scaledY;
        minIm_ += scaledY;

        updateGL();
        event->accept();
    }

    lastPos_ = event->pos();
}

void GLView::drawInfo(int elapsed, int /*kernel*/)
{
    std::vector<QString> info;

    /*info.push_back(QString("View: %1, %2, %3, %4")
        .arg(QString::number(maxRe_)).arg(QString::number(minRe_))
        .arg(QString::number(maxIm_)).arg(QString::number(minIm_)));*/
    //info.push_back(QString("Frame: %1").arg(QString::number(frames_)));
    //info.push_back(QString("Kernel time: %1ms").arg(QString::number(kernel)));
    info.push_back(QString("Max iterations (+/-): %1").arg(QString::number(pbo_.max_iterations())));
    info.push_back(QString("Supersampling (a/d): %1x").arg(QString::number(pow(2, samples_))));
    info.push_back("");
    info.push_back(QString("Frame time: %1ms").arg(QString::number(elapsed)));
    info.push_back(QString("Precision: %1").arg(pbo_.precision().c_str()));

    // Draw stats
	glColor3f(1, 1, 1);
    for(int i = 0; i < info.size(); ++i)
    {
        renderText(10, 20 + i * 16, info[i]);
    }
}

void GLView::drawError(const std::string& what)
{
    glColor3f(1, 0, 0);
    renderText(10, 20, QString("Error: %1").arg(what.c_str()));
    glColor3f(1, 1, 1);
}