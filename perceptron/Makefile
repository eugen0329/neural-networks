SOURCE = main.cpp
TARGET = bin

all:
	g++ -std=c++11 -ggdb $(SOURCE) `pkg-config --cflags opencv` -o $(TARGET) `pkg-config --libs opencv`
