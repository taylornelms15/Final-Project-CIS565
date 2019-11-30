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

#include <iostream>
#include <sys/stat.h>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


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
	if( fileExists(path.c_str()) ){
		return path;
	}


	locations.push_back("/home/slothjet/CIS565/droneMom_ws/src/Final-Project-CIS565/deep_learning_pipeline/data");
	locations.push_back("/home/slothjet/CIS565/droneMom_ws/src/Final-Project-CIS565/deep_learning_pipeline/data/");

	// check each location until the file is found
	const size_t numLocations = locations.size();

	for( size_t n=0; n < numLocations; n++ )
	{
		const std::string str = locations[n] + path;
		if( fileExists(str.c_str()) ){
			return str;
		}
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


// fileExtension
std::string fileExtension( const std::string& path )
{
	std::string ext = path.substr(path.find_last_of(".") + 1);

	transform(ext.begin(), ext.end(), ext.begin(), tolower);

	return ext;
}

