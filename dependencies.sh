#!/usr/bin/env bash
sudo -H apt-get -y install libboost-all-dev python-numpy python-scipy
sudo -H pip3 install --upgrade scikit-learn matplotlib scikit-image dlib lasagne==0.1 theano==0.8 nolearn
