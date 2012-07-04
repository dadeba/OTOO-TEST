#ifndef OTOO_CONFIG_H
#define OTOO_CONFIG_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <map>

namespace OTOO {
  class ConfigFile {
  private:
    std::map<std::string, std::string> hash;
  public:
    ConfigFile() {}
    ~ConfigFile() {}

    void Load(const char *filename) 
    {
      FILE *fp;
      char buf[1000];

      char *name;
      char *value;

      int count = 0;

      fp = fopen(filename, "r");
      if (fp == NULL) {
	std::cerr << "Can't open the config file!!!!!\n";
	exit(-1);
      } 
      while(fgets(buf, 1000, fp) != NULL) {
	//    puts(buf);
	char *p1, *p2;
	if (buf[0] == '#') continue;

	p1 = str_del_space(strtok(buf,  "="));
	p2 = str_del_space(strtok(NULL, "="));
	name  = (char *)malloc(sizeof(char)*2*strlen(p1));;
	value = (char *)malloc(sizeof(char)*2*strlen(p2));;

	strcpy(name, p1);
	strcpy(value, p2);

	hash[std::string(name)] = std::string((char *)value);

	count++;
      }
      fclose(fp);
      std::cerr << "ReadConfigFile : " << count << " lines\n";
    }

    double fGet(std::string p)
    {
      if (hash.find(p) == hash.end()) {
	std::cerr <<"GetConfig :: no such value" << p << "\n";
	exit(-1);
      } else {
	return atof(hash[p].c_str());
      }
    }

    int iGet(std::string p)
    {
      if (hash.find(p) == hash.end()) {
	std::cerr <<"GetConfig :: no such value" << p << "\n";
	exit(-1);
      } else {
	return atoi(hash[p].c_str());
      }
    }

    char *cGet(std::string p)
    {
      if (hash.find(p) == hash.end()) {
	std::cerr <<"GetConfig :: no such value" << p << "\n";
	exit(-1);
      } else {
	return (char *)(hash[p].c_str());
      }
    }

  private:
    char *str_del_space(char *p)
    {
      char *ret;
      
      while(*p++ == ' ');
      ret = --p;
      while(*p != ' ' && *p != 0xa)  p++;
      *p = 0;
      return ret;
    }
  };
}
#endif
