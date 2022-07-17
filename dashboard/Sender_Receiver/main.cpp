#include "MainWindow.h"
#include <QtWidgets/QApplication>
#include "inc/com.h"
#include "QString"

// Recieve

extern Com_Type Com;
extern Ui::MainWindowClass* ui_extern;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    Com_Init(&(Com.ComConfig));

    uint8 data=123;
    const void* ptr = &data;
    ui_extern->output->setText(QString::number(Com_SendSignal(0, ptr)));
        

    return a.exec();
}
