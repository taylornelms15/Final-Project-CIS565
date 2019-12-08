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


	locations.push_back("/home/slothjet/CIS565/droneMoM_ws/src/Final-Project-CIS565/object_detection/data");
	locations.push_back("/home/slothjet/CIS565/droneMoM_ws/src/Final-Project-CIS565/object_detection/data/");
	locations.push_back("/home/taylor/cis565/droneMoM_ws/src/repo/object_detection/data");
	locations.push_back("/home/taylor/cis565/droneMoM_ws/src/repo/object_detection/data/");

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

