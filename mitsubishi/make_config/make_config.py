import glob


problem_number = 1
with open('parameters{}.conf'.format(problem_number), mode='w') as f_w:
    f_w.write('[Problem{}]\n'.format(problem_number))
    conf_files = glob.glob('P{}_*.conf'.format(problem_number))
    for conf_file in conf_files:
        with open(conf_file) as f_r:
            lines = [s.strip() for s in f_r.readlines()]
            if conf_file =='P{}_constants.conf'.format(problem_number):
                for line in lines:
                    line = line.split()
                    f_w.write('{} = {}\n'.format(line[0], line[1]))
            else:
                for i, line in enumerate(lines):
                    lines[i] = float(line)
                if len(lines) == 1:
                    lines = lines[0]
                f_w.write('{} = {}\n'.format(f_r.name[3:-5], lines))
