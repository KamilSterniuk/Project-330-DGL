cmake_minimum_required(VERSION 3.10)
project(my_project)

# Ustawienie ścieżki do katalogu libtorch
set(CMAKE_PREFIX_PATH "C:/libtorch")

find_package(Torch REQUIRED)

add_executable(my_project main.cpp)
target_link_libraries(my_project "${TORCH_LIBRARIES}")
set_property(TARGET my_project PROPERTY CXX_STANDARD 14)

# Blok poniżej jest zalecany dla Windows w celu prawidłowego zarządzania DLL-ami
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET my_project
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:my_project>)
endif (MSVC)
