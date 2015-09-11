#pragma once 
#include <iostream>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>
using namespace std;


/*
	The Log class is used to record values for plotting and debugging.
*/
class Log
{
private:
	//Index variables
	unsigned long t = 0;
	unsigned long i = 0;
	//Containers
	map<string,vector<pair<unsigned long,float>>> data_;
	//Functions
	Log() {}
	Log(Log const&)				= delete;
	void operator=(Log const&)	= delete;
public:
	//Get the log instance
	static Log& instance()
	{
		static Log instance;
		return instance;
	}
	//Record the value of a variable
	void record(std::string var, float val)
	{
		auto search = data_.find(var);
		if(search != data_.end())
			search->second.push_back(make_pair(t,val));
		else
			data_.insert(make_pair(var, vector<pair<unsigned long,float>>{make_pair(t,val)} ));
	}
	//Next time step
	void next()
	{
		t++;
	}
	//Write log to disk
	void dump(string file)
	{
		//Anounce dump
		cout << "[!] DUMPING LOG" << endl;

		//File stream
		ofstream out(file);

		//Build a string vector
		vector<vector<string>> format(t+2,vector<string>(data_.size()+1,""));

		//Write time
		format[0][0] = "time";
		for(int i=1; i<t+2; i++)
			format[i][0] = to_string(i-1);

		//Write data
		int index = 1;
		for(auto itr = data_.begin(); itr != data_.end(); itr++)
		{
			//Header
			format[0][index] = itr->first;
			//Source
			for(auto &p : itr->second)
				format[p.first+1][index] = to_string(p.second);
			index++;
		}

		//Write to file
		for(auto &r : format)
		{
			for(auto &c : r)
				out << c << ",";
			out << endl;
		}

		//Reset
		data_.clear();
		i++;
		t = 0;
	}
};
