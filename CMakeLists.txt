cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed. "
        "Please create a separate build directory and run cmake from there.\n"
        "To clean up, remove CMakeCache.txt and CMakeFiles/ directory.")
endif()

project(zenzai-plus-Hazkey-Fcitx5 VERSION 0.3.0)
include(GNUInstallDirs)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

add_subdirectory(fcitx5-hazkey)
add_subdirectory(hazkey-server)
add_subdirectory(hazkey-settings)

install(DIRECTORY distribution/dictionaries/ DESTINATION "${CMAKE_INSTALL_DATADIR}/hazkey/dictionaries")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/zenzai-plus-Hazkey-Fcitx5")
install(DIRECTORY distribution/licenses/ DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/zenzai-plus-Hazkey-Fcitx5/licenses")
