set (CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7-a)

set(TOOLCHAIN_PREFIX arm-linux-gnueabihf)
set(RK_SDK "/opt/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/")



# specify the cross compiler
SET(CMAKE_C_COMPILER   ${RK_SDK}/bin/${TOOLCHAIN_PREFIX}-gcc)
SET(CMAKE_ASM_COMPILER   ${CMAKE_C_COMPILER})
SET(CMAKE_CXX_COMPILER   ${RK_SDK}/bin/${TOOLCHAIN_PREFIX}-g++)
SET(CMAKE_AR_COMPILER   ${RK_SDK}/bin/${TOOLCHAIN_PREFIX}-ar)
SET(CMAKE_RANLIB_COMPILER   ${RK_SDK}/bin/${TOOLCHAIN_PREFIX}-ranlib)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_FIND_ROOT_PATH ${RK_SDK})


# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

