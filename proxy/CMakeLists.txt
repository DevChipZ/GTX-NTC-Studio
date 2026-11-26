cmake_minimum_required(VERSION 3.16)
project(gtx_ntc_dxgi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
add_library(dxgi SHARED dxgi_proxy.cpp dxgi.def)
