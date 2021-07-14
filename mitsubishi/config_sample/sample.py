import configparser
import os


def show_config(ini):
    for section in ini.sections():
        print('[%s]' % section)
        for key in ini.options(section):
            print('%s.%s =%s' % (section, key, ini.get(section, key)))
    return


params = configparser.ConfigParser()
params.read("./param.conf")

print("name", params)
show_config(params)

show_config(params)
sigma = params.getint('CMAES', 'sigma') + 1
print(sigma)
dic = dict(params.items('Nelder-Mead'))
print(dic)
