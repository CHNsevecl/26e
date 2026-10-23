#include "uart.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstdio>

//类初始化
UART::UART(const std::string& device, int baudrate)
    : fd_(-1),
      device_(device),
      baudrate_(baudrate)
{
}



//析构函数
UART::~UART()
{
    close();
}



bool UART::open()
{
    //打开串口
    fd_ = ::open(device_.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0) {
        std::perror(("打开串口 " + device_).c_str());
        return false;
    }

    struct termios tty;
    if (tcgetattr(fd_, &tty) != 0) { // tcgetattr(fd_, &tty)将fd_的配置修改进tty结构体中（传入的是tty的地址）
        std::perror(("读取串口配置 " + device_).c_str());
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    // 设置为原始模式，避免Linux把数据当作终端指令进行处理
    cfmakeraw(&tty);

    // 设置波特率
    switch (baudrate_) {
        case 9600:
            cfsetispeed(&tty, B9600);
            cfsetospeed(&tty, B9600);
            break;
        case 19200:
            cfsetispeed(&tty, B19200);
            cfsetospeed(&tty, B19200);
            break;
        case 38400:
            cfsetispeed(&tty, B38400);
            cfsetospeed(&tty, B38400);
            break;
        case 57600:
            cfsetispeed(&tty, B57600);
            cfsetospeed(&tty, B57600);
            break;
        case 115200:
            cfsetispeed(&tty, B115200);
            cfsetospeed(&tty, B115200);
            break;
        default:
            std::fprintf(stderr, "不支持的串口波特率：%d\n", baudrate_);
            ::close(fd_);
            fd_ = -1;
            return false;
    }

    // 8 个数据位
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    // 无校验
    tty.c_cflag &= ~PARENB;

    // 1 个停止位
    tty.c_cflag &= ~CSTOPB;

    // 关闭硬件流控
    tty.c_cflag &= ~CRTSCTS;

    // 开启接收，本地模式
    tty.c_cflag |= CREAD | CLOCAL;

    // 应用串口配置
    if (tcsetattr(fd_, TCSANOW, &tty) != 0)
    {
        std::perror(("应用串口配置 " + device_).c_str());
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    return true;
}



/**
 * @brief 向连接发送数据
 * @param data  指向待发送数据的缓冲区指针（输入参数）。
 * @param length 要发送的数据字节数。必须大于 0。         
* @return true  数据发送成功。
 */
bool UART::send(const uint8_t* data, size_t length)
{
    if (fd_ < 0)
    {
        return false;
    }

    if (data == nullptr || length == 0)
    {
        return false;
    }

    size_t total_sent = 0;

    while (total_sent < length)
    {
        ssize_t n = ::write(
            fd_,
            data + total_sent,
            length - total_sent
        );

        if (n < 0)
        {
            return false;
        }

        total_sent += static_cast<size_t>(n);
    }

    return true;
}


/**
 * @brief 向连接发送字符串数据
 * @param data  要发送的字符串
 * @return true  数据发送成功
 */
bool UART::send(const std::string& data)
{
    return send(
        reinterpret_cast<const uint8_t*>(data.data()), // reinterpret_cast<const uint8_t*>将某一种指针以uint8_t*类型的指针进行转换，reinterpret_cast<const uint8_t*>(data.data())将字符串数据转换为 const uint8_t* 类型的指针，以便与 send 函数的参数类型匹配。
        data.size()                                    // data.size()返回字符串的长度，即要发送的数据字节数。
    );
}


/**
 * @brief 从连接接收数据
 * @param buffer  指向接收数据的缓冲区指针（输出参数）。
 * @param length  要接收的数据字节数。必须大于 0。
 * @param timeout_ms  接收超时时间，单位为毫秒。默认值为 -1，表示无限等待。
 * @return 接收到的数据字节数，若接收超时或发生错误，则返回 -1。
 */
int UART::receive(uint8_t* buffer, size_t length, int timeout_ms)
{
    if (fd_ < 0)
    {
        return -1;
    }

    if (buffer == nullptr || length == 0)
    {
        return -1;
    }

    if (timeout_ms < 0)
    {
        ssize_t n = ::read(fd_, buffer, length);

        if (n < 0)
        {
            return -1;
        }

        return static_cast<int>(n);
    }

    fd_set readfds;
    FD_ZERO(&readfds);        // 清除readfds集合中的所有文件描述符
    FD_SET(fd_, &readfds);    // 将fd_添加到readfds集合中，表示我们希望监视该文件描述符的可读性。

    struct timeval timeout;   // Linux 用来表示时间长度的结构体。
    timeout.tv_sec = timeout_ms / 1000;     // 将超时时间转换为秒数，timeout_ms / 1000表示将毫秒转换为秒。
    timeout.tv_usec = (timeout_ms % 1000) * 1000;   // 将超时时间转换为微秒数，timeout_ms % 1000表示获取毫秒部分，然后乘以1000将其转换为微秒。

    int ret = select(fd_ + 1, &readfds, nullptr, nullptr, &timeout);    
    //fd_ + 1规定了select函数监视的文件描述符范围
    //&readfds表示我们希望监视的可读文件描述符集合，也就是说readfds会决定只关注哪些设备，select只是搭建总线用的
    //nullptr表示我们不关心可写和异常文件描述符集合
    //&timeout表示超时时间。

    if (ret < 0)
    {
        return -1;
    }

    if (ret == 0)
    {
        return 0;
    }

    ssize_t n = ::read(fd_, buffer, length);

    if (n < 0)
    {
        return -1;
    }

    return static_cast<int>(n);
}

void UART::close()
{
    if (fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }
}

bool UART::isOpen() const
{
    return fd_ >= 0;
}
