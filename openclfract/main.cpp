//
//  Author  : Matti Määttä
//  Summary : Application entry point
//

#include "glview.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    GLView view;
    view.show();

    return a.exec();
}
