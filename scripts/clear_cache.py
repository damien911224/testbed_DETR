
import subprocess
import os, time


sh_file_path = os.path.join('clear_cache.sh')
clear_term_in_secs = 60.0 * 1.0

cmd = 'sh {}'.format(sh_file_path)


if __name__ == '__main__':

    while True:


        p = subprocess.call('{}'.format(cmd), shell=True)
        print 'Clear Cache ..'
        print '==============================================================================================='
        os.system('free -h')
        print '==============================================================================================='
        time.sleep(clear_term_in_secs)
