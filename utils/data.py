def retrieve_param(file_path: str = 'data/parameters.txt'):
    with open(file_path, 'r') as file :
        commands = file.readlines()

    w_str = ''
    str_command = ''
    for i, command in enumerate(commands):
        if i > 1 or i != len(commands) + 1 :
            w_str += command
        else :
            str_command += command

    return w_str + str_command
