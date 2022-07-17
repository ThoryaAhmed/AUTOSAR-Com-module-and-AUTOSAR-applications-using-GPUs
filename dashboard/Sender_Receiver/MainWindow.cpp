#include "MainWindow.h"
#include <QHostAddress>
#include <QTcpSocket>
#include <QAbstractSocket>
#include <QDebug>
#include "Platform_Types.h"
#include "inc/Com.h"
#include <string>
#include "cuda_code.cuh"

Ui::MainWindowClass* ui_extern;
extern uint8 ComIPdu0Buffer[2];
extern uint8 ComIPdu1Buffer[2];
extern Com_Type Com;
Com_SignalType* ComSignalLocal;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui_extern = &ui;
    ui_extern->setupUi(this);
    ui.setupUi(this);
    _server.listen(QHostAddress::Any, 4242);
    connect(&_server, SIGNAL(newConnection()), this, SLOT(onNewConnection()));

    ui.larrow_b->setVisible(0);
    ui.rarrow_b->setVisible(0);
    ui.fuel_b->setVisible(0);
    ui.battery_b->setVisible(0);
    ui.opendoor_b->setVisible(0);
    ui.light_b->setVisible(0);
    ui.fix_b->setVisible(0);
    
}

MainWindow::~MainWindow()
{}
void MainWindow::onNewConnection()
{
    QTcpSocket* clientSocket = _server.nextPendingConnection();
    connect(clientSocket, SIGNAL(readyRead()), this, SLOT(onReadyRead()));
    connect(clientSocket, SIGNAL(stateChanged(QAbstractSocket::SocketState)), this, SLOT(onSocketStateChanged(QAbstractSocket::SocketState)));

    _sockets.push_back(clientSocket);
    for (QTcpSocket* socket : _sockets) {
        socket->write(QByteArray::fromStdString(clientSocket->peerAddress().toString().toStdString() + " connected to server !\n"));
    }
}

void MainWindow::onSocketStateChanged(QAbstractSocket::SocketState socketState)
{
    if (socketState == QAbstractSocket::UnconnectedState)
    {
        QTcpSocket* sender = static_cast<QTcpSocket*>(QObject::sender());
        _sockets.removeOne(sender);
    }
}

void MainWindow::onReadyRead()
{
    QTcpSocket* sender = static_cast<QTcpSocket*>(QObject::sender());
    QByteArray datas = sender->readAll();
    //QString data = QString(datas);
    //ui.statusbar->showMessage(datas);
    //uint8 len = datas.length();
    int ipdu_id = datas.mid(1,1).toInt();
    uint8 Signaldata1 = static_cast<uint8>(datas.at(0));
    //int Signaldata1 = datas.mid(0,1);
    //uint8(Signaldata1.data());
    //int Signaldata = Signaldata1.toInt();
    uint8 signalID = datas.mid(2).toInt();
    //QString myString(datas);
    //QStringRef ipdu_id(&myString, 0, 1); // subString contains "is"
    //QStringRef data(&myString, 1, 1); // subString contains "is"

    ui.statusbar->showMessage(QString(Signaldata1) + ipdu_id);
    //ui.fuel_b->setVisible(2);
    /// <summary>
    ///  QByte to array
    /// </summary>
    /// com_receive_signal(signal_id,ptr to data)
    /// 
    QString DataAsString = datas;
    uint8 signal = 0;
    
    if (DataAsString[0] == 'k');

    //// choose the ipdu depend on the data
    else if (ipdu_id == 0) {
        //memcpy(ComIPdu0Buffer, &data, DataAsString.size());
        ComSignalLocal = Com.ComConfig.ComIPdu[ipdu_id].ComIPduSignalRef[signalID];
        *(ComSignalLocal->ComBufferRef) =  Signaldata1;
        if (signalID==0) {
            
            Com_ReceiveSignal_GPU(signalID, &signal);
            ui.fuel_b->setVisible(signal);
        }
         if (signalID==1) {

            Com_ReceiveSignal_GPU(1, &signal);
            ui.opendoor_b->setVisible((signal));

            //ui.battery_b->setVisible(DataAsString.mid(1).toInt());
        }
        else if (signalID==2) {

            Com_ReceiveSignal_GPU(2, &signal);
            ui.light_b->setVisible((signal));

            //ui.light_b->setVisible(DataAsString.mid(1).toInt());
        }
        else if (signalID==3) {

            Com_ReceiveSignal_GPU(3, &signal);
            ui.battery_b->setVisible((signal));

            //ui.opendoor_b->setVisible(DataAsString.mid(1).toInt());
        }
        else if (signalID==4) {

            Com_ReceiveSignal_GPU(4, &signal);
            ui.fix_b->setVisible((signal));

            //ui.fuel_b->setVisible(DataAsString.mid(1).toInt());
        }
        else if (signalID==5)
        {
            Com_ReceiveSignal_GPU(5, &signal);
            ui.larrow_b->setVisible((signal));
            //ui.rarrow_b->setVisible(DataAsString.mid(1).toInt());
        }
        else if (signalID==6)
        {
            Com_ReceiveSignal_GPU(6, &signal);
            ui.rarrow_b->setVisible((signal));
            //ui.larrow_b->setVisible(DataAsString.mid(1).toInt());
        }

    }

    else if (ipdu_id == 1) {
        memcpy(ComIPdu1Buffer, &Signaldata1, DataAsString.size());
        //ComSignalLocal = Com.ComConfig.ComIPdu[ipdu_id].ComIPduSignalRef[signalID];
        //*(ComSignalLocal->ComBufferRef) = Signaldata1;
        if (signalID==7) {
            Com_ReceiveSignal_GPU(7, &signal);
            ui.speed->setText(QString::number(signal));
        }

        else if (signalID==8) {
            Com_ReceiveSignal_GPU(8, &signal);
            ui.degree->setText(QString::number(signal));
        }
    }

    for (QTcpSocket* socket : _sockets) {
        if (socket != sender)
            socket->write(QByteArray::fromStdString(sender->peerAddress().toString().toStdString() + ": " + datas.toStdString()));
    }
}

void MainWindow::on_larrow_c_stateChanged(int arg1)
{
    ui.larrow_b->setVisible(arg1);
}


void MainWindow::on_rarrow_c_stateChanged(int arg1)
{
    ui.rarrow_b->setVisible(arg1);
}


void MainWindow::on_fuel_c_stateChanged(int arg1)
{
    ui.fuel_b->setVisible(arg1);
}


void MainWindow::on_opendoor_c_stateChanged(int arg1)
{
    ui.opendoor_b->setVisible(arg1);
}


void MainWindow::on_light_c_stateChanged(int arg1)
{
    ui.light_b->setVisible(arg1);
}


void MainWindow::on_battery_c_stateChanged(int arg1)
{
    ui.battery_b->setVisible(arg1);
}


void MainWindow::on_fix_c_stateChanged(int arg1)
{
    ui.fix_b->setVisible(arg1);
}
void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    QString s = QString::number(value);
    ui.speed->setText(s);
}


void MainWindow::on_horizontalSlider_2_valueChanged(int value)
{
    QString s = QString::number(value);
    ui.degree->setText(s);
}

