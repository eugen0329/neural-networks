SOURCE = main.cpp
TARGET = bin

all:
	g++ -std=c++14 $(SOURCE)  $(BOOST_FLAGS) `pkg-config --cflags opencv` -o $(TARGET) `pkg-config --libs opencv`
