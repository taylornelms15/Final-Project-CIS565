#ifndef __FILESYSTEM_UTIL_H__
#define __FILESYSTEM_UTIL_H__

#include <string>
#include <vector>



std::string locateFile( const std::string& path );

std::string locateFile( const std::string& path, std::vector<std::string>& locations );


bool fileExists( const char* path, bool regularFilesOnly=false );


std::string fileExtension( const std::string& path );


#endif
