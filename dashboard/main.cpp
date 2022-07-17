#include "MainWindow.h"
#include <QtWidgets/QApplication>
#include "cudacode.cuh"
#include <bits/stdc++.h>
#include "inc/com.h"
#include "QString"


// Send

extern Com_Type Com;
//extern Ui::MainWindowClass* ui_extern;


using namespace std;


int main(int argc, char *argv[])
{

	QApplication a(argc, argv);
    MainWindow w;
    w.show();

    Com_Init(&(Com.ComConfig));

   // uint8 data = 123;
   // const void* ptr = &data;
    
   wrapper();

    return a.exec();
}
