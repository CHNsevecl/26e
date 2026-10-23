#ifndef uart_hpp
#define uart_hpp

#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

class UART{
public:
    UART(const std::string& device,int baudrate);//构造函数

    ~UART();//析构函数

    bool open();
    void close();

    bool isOpen() const;

    bool send(const uint8_t* data, size_t length);
    bool send(const std::string& data);

    int receive(uint8_t* buffer,size_t length,int timeout_ms = -1);

private:
    int fd_;

    std::string device_;
    int baudrate_;
};

#endif 