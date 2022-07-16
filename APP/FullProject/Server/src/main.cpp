#include "SocketsServer.h"
#include <QtWidgets/QApplication>

#include "Com.cuh"

extern Com_Type Com_CPU[1];

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    SocketsServer w;
    w.show();
    Com_Init();
    return a.exec();
}
