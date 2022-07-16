#include "SocketsClient.h"
#include <QtWidgets/QApplication>

// #include "./gen/Com_Cfg.cuh"
// #include "./inc/Com_Types.cuh"
#include "Com.cuh"

extern Com_Type Com_CPU[1];

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    SocketsClient w;
    w.show();
    Com_Init();
    return a.exec();
}
