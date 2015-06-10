cmake_minimum_required(VERSION 2.4.6)
SET(PROJECT_NAME briskcvd)
PROJECT(${PROJECT_NAME})

#set dependancies
FIND_PACKAGE(agast REQUIRED)

file(GLOB BRISK_SOURCE_FILES  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
file(GLOB BRISK_HEADER_FILES  "${CMAKE_CURRENT_SOURCE_DIR}/include/brisk/*.h")

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}/include/brisk/
	${agast_INCLUDE_DIRS})

add_definitions(-fPIC -msse2 -mssse3 -march=nocona -Wextra -Winit-self -Woverloaded-virtual -Wnon-virtual-dtor -Wsign-promo -Wno-long-long )

#build the library dynamic and static versions
add_library(briskcvd SHARED ${BRISK_SOURCE_FILES} ${BRISK_HEADER_FILES})

################################################################################
# Create the briskcvdConfig.cmake file for other cmake projects.
GET_TARGET_PROPERTY(FULL_LIBRARY_NAME ${PROJECT_NAME} LOCATION )
SET(briskcvd_LIBRARIES ${FULL_LIBRARY_NAME} )
SET(briskcvd_LIBRARY_DIR ${PROJECT_BINARY_DIR} )
SET(briskcvd_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/briskcvd.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/briskcvdConfig.cmake @ONLY IMMEDIATE )
export(PACKAGE briskcvd)

INSTALL(DIRECTORY include/brisk DESTINATION ${CMAKE_INSTALL_PREFIX}/include FILES_MATCHING PATTERN "*.h")
INSTALL(DIRECTORY include/brisk DESTINATION ${CMAKE_INSTALL_PREFIX}/include FILES_MATCHING PATTERN "*.hpp")
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

# Need to add Here compile instruction for demo