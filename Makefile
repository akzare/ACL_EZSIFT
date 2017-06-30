#
#	make file
#


.SUFFIXES : .cpp .o 

# 	GNU C++ compiler
CC = g++

TARGET      = $(notdir $(shell pwd))	## current foldername is target name

BUILD_PATH 	= ./build

BIN_PATH 	= $(BUILD_PATH)/bin
OBJ_PATH 	= $(BUILD_PATH)/obj

SRC_PATH 	= .

SRCS		= $(wildcard $(SRC_PATH)/*.cpp)
OBJS 		= $(SRCS:$(SRC_PATH)/%.cpp=$(OBJ_PATH)/%.o)

INCS	+= -I.
INCS	+= -I/usr/local/include
INCS	+= -I/home/odroid/acl/ComputeLibrary-master
INCS	+= -I/home/odroid/acl/ComputeLibrary-master/include

CXXFLAGS = $(INCS) -c -O2 -std=c++11 -mcpu=cortex-a8 -mfpu=neon #-DARM_COMPUTE_CL=1 #-fpermissive #-W -Wall -O0
	
# 	Link Options
LIBS		+= -lpthread

LIBS		+= -larm_compute
LIBS		+= -lOpenCL

#LIVDIRS     += -L$(OPENCV_LIB_PATH)
LIVDIRS     += -L/home/odroid/acl/ComputeLibrary-master/build
LIBS 		+= -lopencv_contrib
LIBS 		+= -lopencv_core
LIBS 		+= -lopencv_highgui
LIBS 		+= -lopencv_imgproc
LIBS 		+= -lopencv_legacy
LIBS 		+= -lopencv_video
LIBS 		+= -lopencv_videostab

LDFLAGS     = $(LIVDIRS) -lm $(LIBS)

#	rm options
RM 			= @rm -rfv

# 	mkdir options
MKDIR 		= @mkdir -p

$(BIN_PATH)/$(TARGET): $(OBJS)
	$(MKDIR) $(BIN_PATH)
	$(CC) -o $(BIN_PATH)/$(TARGET) $(OBJS) $(LDFLAGS)

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(MKDIR) $(OBJ_PATH)
	$(CC) $(CXXFLAGS) $< -o $@

all : $(BIN_PATH)/$(TARGET)

#	dependency
dep :
	$(MKDIR) $(BUILD_PATH)
	$(CC) -M $(INCS) $(SRCS) > $(BUILD_PATH)/.depend

#	clean
clean:
	$(RM) $(BUILD_PATH)
	$(RM) $(TARGET)
	@echo "Done."

#	include dependency
ifeq (.depend,$(wildcard .depend))
include .depend
endif

