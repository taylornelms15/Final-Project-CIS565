#!/bin/bash

EIGEN_VERSION=3.3.7

mkdir /tmp/dl && cd /tmp/dl

echo "Downloading installer files"
curl -s -L -o eigen-$EIGEN_VERSION.tar.gz http://bitbucket.org/eigen/eigen/get/$EIGEN_VERSION.tar.gz

echo "Installing Eigen version $EIGEN_VERSION"
tar xf eigen-$EIGEN_VERSION.tar.gz
cd eigen-eigen-*
mkdir build && cd build
cmake ..
sudo make install
