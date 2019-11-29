import fileinput
import shutil
import inspect
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)


class Param():
    def __init__(self, param_path="param.py"):
        self.param_path = param_path
        shutil.copy(self.param_path, 'param_copy.py')

    def update(self, *args):
        dic_to_update = {}
        for var in args:
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            name = [var_name for var_name,
                    var_val in callers_local_vars if var_val is var]
            dic_to_update[name[0]] = var
        print(dic_to_update)
        self.update_dic(dic_to_update)

    def update_dic_(self, dic_param_to_update):
        with fileinput.input(self.param_path, inplace=True) as param_file:
            for line in param_file:
                for key, value in dic_param_to_update.items():
                    line = line.strip('\n')
                    # if we find the line in which
                    # the value of the parameter is given
                    if (line.find(key+'=') == 0) or (line.find(key+' =') == 0):
                        # to change a string parameter, we must add quotemarks
                        if isinstance(value, str):
                            value = "'"+value+"'"
                        # then we replace that line with a new one
                        line = key+'='+str(value)
                # we let other lines like they used to be
                sys.stdout.buffer.write(line.encode('utf-8'))

    def update_dic(self, dic_param_to_update):
        file_path = self.param_path
        for key, value in dic_param_to_update.items():
            #Create temp file
            fh, abs_path = mkstemp()
            with fdopen(fh,'w', encoding="utf-8") as new_file:
                with open(file_path, encoding="utf-8") as old_file:
                    for line in old_file:
                        # if we find the line in which
                        # the value of the parameter is given
                        if (line.find(key+'=') == 0) or (line.find(key+' =') == 0):
                            # to change a string parameter, we must add quotemarks
                            if isinstance(value, str):
                                value = "'"+value+"'"
                            # then we replace that line with a new one
                            newline = key+'='+str(value)+'\n'
                        # we let other lines like they used to be
                        else:
                            newline = line
                        new_file.write(newline)                     
            #Remove original file
            remove(file_path)
            #Move new file
            move(abs_path, file_path)

                    
    def reset(self):
        shutil.move('param_copy.py', 'param.py')

    @staticmethod
    def retrieve_name(var):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return [var_name for var_name,
                var_val in callers_local_vars if var_val is var]
