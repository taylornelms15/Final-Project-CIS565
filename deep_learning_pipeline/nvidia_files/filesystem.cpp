/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "filesystem.h"

#include <sys/stat.h>
#include <algorithm>

// ExecutablePath
static std::string ExecutablePath()
{
	char buf[512];
	memset(buf, 0, sizeof(buf));	// readlink() does not NULL-terminate

	const ssize_t size = readlink("/proc/self/exe", buf, sizeof(buf));

	if( size <= 0 )
		return "";
	
	return std::string(buf);
}


// ExecutableDirectory
static std::string ExecutableDirectory()
{
	const std::string path = ExecutablePath();

	if( path.length() == 0 )
		return "";

	return filePath(path);
}


// WorkingDirectory
static std::string WorkingDirectory()
{
	char buf[1024];

	char* str = getcwd(buf, sizeof(buf));

	if( !str )
		return "";

	return buf;
}

// absolutePath
std::string absolutePath( const std::string& relative_path )
{
	const std::string proc = ExecutableDirectory();
	std::cout << "file " << proc << ": " proce + relative_path << std::endl;
	return proc + relative_path;
}


// locateFile
std::string locateFile( const std::string& path )
{
	std::vector<std::string> locations;
	return locateFile(path, locations);
}


// locateFile
std::string locateFile( const std::string& path, std::vector<std::string>& locations )
{
	// check the given path first
	if( fileExists(path.c_str()) )
		return path;

	// add standard search locations
	locations.push_back(ExecutableDirectory());

	std::cout << "ExecutableDirectory " << ExecutableDirectory() << std::endl;

	locations.push_back("/usr/local/bin/");
	locations.push_back("/usr/local/");
	locations.push_back("/opt/");

	locations.push_back("images/");
	locations.push_back("data/");
	locations.push_back("/usr/local/bin/images/");
	locations.push_back("~/CIS565/droneMoM/src/FinalProjectCIS565/ros_deep_learning/");

	// check each location until the file is found
	const size_t numLocations = locations.size();

	for( size_t n=0; n < numLocations; n++ )
	{
		const std::string str = locations[n] + path;

		if( fileExists(str.c_str()) )
			return str;
	}

	return "";
}



// fileExists
bool fileExists( const char* path, bool regularFilesOnly )
{
	if( !path )
		return false;

	struct stat fileStat;
	const int result = stat(path, &fileStat);

	if( result == -1 )
	{
		//printf("%s does not exist.\n", path);
		return false;
	}

	if( !regularFilesOnly )
		return true;

	if( S_ISREG(fileStat.st_mode) )
		return true;
	
	return false;
}


// fileSize
size_t fileSize( const char* path )
{
	if( !path )
		return 0;

	struct stat fileStat;

	const int result = stat(path, &fileStat);

	if( result == -1 )
	{
		printf("%s does not exist.\n", path);
		return 0;
	}

	//printf("%s  size %zu bytes\n", path, (size_t)fileStat.st_size);
	return fileStat.st_size;
}


// filePath
std::string filePath( const std::string& filename )
{
	const std::string::size_type slashIdx = filename.find_last_of("/");

	if( slashIdx == std::string::npos || slashIdx == 0 )
		return filename;

	return filename.substr(0, slashIdx + 1);
}


// fileExtension
std::string fileExtension( const std::string& path )
{
	std::string ext = path.substr(path.find_last_of(".") + 1);

	transform(ext.begin(), ext.end(), ext.begin(), tolower);

	return ext;
}


// fileRemoveExtension
std::string fileRemoveExtension( const std::string& filename )
{
	const std::string::size_type dotIdx   = filename.find_last_of(".");
	const std::string::size_type slashIdx = filename.find_last_of("/");

    if( dotIdx == std::string::npos )
		return filename;

	if( slashIdx != std::string::npos && dotIdx < slashIdx )
		return filename;

    return filename.substr(0, dotIdx);
}


// fileChangeExtension
std::string fileChangeExtension(const std::string& filename, const std::string& newExtension)  
{
	return fileRemoveExtension(filename).append(newExtension);
}


// processPath
std::string processPath()
{
	return ExecutablePath();
}


// processDirectory
std::string processDirectory()
{
	return ExecutableDirectory();
}


// workingDirectory
std::string workingDirectory()
{
	return WorkingDirectory();
}
