from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars
from distutils.command.build_clib import build_clib
from distutils.command.build_ext import build_ext
from distutils.command.install_headers import install_headers as install_headers_orig
import json
import pybind11
import pkgconfig
import os
import os.path
import sys
from os.path import join
import tmci

def find_in_path(name, path):
    """
    Find a file in a search path
    """
    
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """
    Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """
    
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = join(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))
    cudaconfig = {'home': home, 'nvcc': nvcc, 'include': join(home, 'include'), 'lib64': join(home, 'lib64')}
    
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


class install_headers(install_headers_orig):
    """The install_headers command is redefined to
    remove the "python3.Xm" in the installation path"""

    def run(self):
        a = self.install_dir.split('/')
        if 'python' in a[-2]:
            del a[-2]
        # delete also the last "flamestore" subdir
        del a[-1]
        self.install_dir = '/'.join(a)
        headers = self.distribution.headers or []
        for header in headers:
            self.mkpath(self.install_dir)
            (out, _) = self.copy_file(header, self.install_dir)
            if 'swift' in header:
                new_out = out.replace('.h','.swift')
                os.rename(out, new_out)
                out = new_out
            self.outfiles.append(out)

src_dir = os.path.dirname(os.path.abspath(__file__)) + '/flamestore/src'

tf_info = {
        'libraries'      : None,
        'library_dirs'   : None,
        'include_dirs'   : None,
        'extra_cxxflags' : None
        }
try:
    with open('tensorflow.json') as f:
        tf_info = json.loads(f.read())
except:
    try:
        import tensorflow as tf
        tf_info['include_dirs'] = [tf.sysconfig.get_include()]
        tf_info['library_dirs'] = [tf.sysconfig.get_lib()]
        tf_info['extra_cxxflags'] = tf.sysconfig.get_compile_flags()
        tf_info['libraries'] = [tf.sysconfig.get_link_flags()[-1][2:]]
    except:
        pass
print(tf_info)

cxxflags = tf_info['extra_cxxflags']

def get_pybind11_include():
    path = os.path.dirname(pybind11.__file__)
    return '/'.join(path.split('/')[0:-4] + ['include'])

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(flag for flag in opt.split() if flag != '-Wstrict-prototypes')

CUDA = locate_cuda()
thallium     = pkgconfig.parse('thallium')
#spdlog       = pkgconfig.parse('spdlog')
#bake_client  = pkgconfig.parse('bake-client')
#bake_server  = pkgconfig.parse('bake-server')
#sdskv_client = pkgconfig.parse('sdskv-client')
#sdskv_server = pkgconfig.parse('sdskv-server')
jsoncpp      = pkgconfig.parse('jsoncpp')
ssg          = pkgconfig.parse('ssg')

flamestore_server_module_libraries    = thallium['libraries'] + tf_info['libraries'] + jsoncpp['libraries'] + ssg['libraries']
flamestore_server_module_library_dirs = thallium['library_dirs'] + tf_info['library_dirs'] +  jsoncpp['library_dirs'] + ssg['library_dirs']
flamestore_server_module_include_dirs = thallium['include_dirs'] + tf_info['include_dirs'] + jsoncpp['include_dirs'] + ssg['include_dirs']
flamestore_server_module = Extension('_flamestore_server',
        ['flamestore/src/server/backend.cpp',
         'flamestore/src/server/memory_backend.cpp',
         #'flamestore/src/server/mochi_backend.cpp',
         'flamestore/src/server/master_server.cpp',
         #'flamestore/src/server/storage_server.cpp',
        # 'flamestore/src/server/mmapfs_backend.cpp',
        # 'flamestore/src/server/provider.cpp',
         'flamestore/src/server/server_module.cpp'
        ],
        libraries=flamestore_server_module_libraries,
        library_dirs=flamestore_server_module_library_dirs,
        include_dirs=flamestore_server_module_include_dirs + ['/home/mmadhya1/git/spack-cooley/var/spack/environments/fl2/.spack-env/view/lib/python3.9/site-packages/pybind11/include/pybind11/include', '/home/mmadhya1/git/spack-cooley/var/spack/environments/fl2/.spack-env/view/lib/python3.9/site-packages/pybind11/include'],
        extra_compile_args=cxxflags,
        depends=[])

flamestore_client_module_libraries    = thallium['libraries']       \
                                      + tf_info['libraries']        \
				      + jsoncpp['libraries']     \
				      + ssg['libraries'] 	\
                                      + [ ':'+tmci.get_library() ] + [CUDA['lib64']]
flamestore_client_module_library_dirs = thallium['library_dirs']    \
                                      + tf_info['library_dirs']     \
				      + jsoncpp['library_dirs']     \
				      + ssg['library_dirs']         \
                                      + [ tmci.get_library_dir() ] + [CUDA['lib64']]
flamestore_client_module_include_dirs = thallium['include_dirs']    \
                                      + [ src_dir ]                 \
				      + jsoncpp['include_dirs']     \
				      + ssg['include_dirs'] 	    \
                                      + tf_info['include_dirs'] + [CUDA['include']]
flamestore_client_module = Extension('_flamestore_client',
        ['flamestore/src/client/client.cpp',
         'flamestore/src/client/client_module.cpp',
         'flamestore/src/client/tmci_backend.cpp'],
        libraries=flamestore_client_module_libraries,
        library_dirs=flamestore_client_module_library_dirs,
        runtime_library_dirs=flamestore_client_module_library_dirs,
        include_dirs=flamestore_client_module_include_dirs + ['/home/mmadhya1/git/spack-cooley/var/spack/environments/fl2/.spack-env/view/lib/python3.9/site-packages/pybind11/include'], 
        extra_compile_args=cxxflags,
        depends=[])

flamestore_admin_module_libraries     = thallium['libraries']  + tf_info['libraries'] + jsoncpp['libraries'] + ssg['libraries'] 
flamestore_admin_module_library_dirs = thallium['library_dirs'] + tf_info['library_dirs'] + jsoncpp['library_dirs'] + ssg['library_dirs']
flamestore_admin_module_include_dirs = thallium['include_dirs'] + tf_info['include_dirs'] +  jsoncpp['include_dirs'] + ssg['include_dirs'] \
                                      + [ src_dir ]
flamestore_admin_module = Extension('_flamestore_admin',
        ['flamestore/src/admin/admin.cpp',
         'flamestore/src/admin/admin_module.cpp'],
        libraries=flamestore_admin_module_libraries,
        library_dirs=flamestore_admin_module_library_dirs,
        runtime_library_dirs=flamestore_admin_module_library_dirs,
        include_dirs=flamestore_admin_module_include_dirs + ['/home/mmadhya1/git/spack-cooley/var/spack/environments/fl2/.spack-env/view/lib/python3.9/site-packages/pybind11/include', '/home/mmadhya1/git/spack-cooley/var/spack/environments/fl2/.spack-env/view/lib/python3.9/site-packages/pybind11/include'], 

        extra_compile_args=cxxflags,
        depends=[])

setup(name='flamestore',
      version='0.3',
      author='Matthieu Dorier',
      description='''Mochi service to store and load tensorflow models''',
      ext_modules=[ flamestore_client_module,
                    flamestore_server_module,
                    flamestore_admin_module,
                  ],
      packages=['flamestore'],
      scripts=['bin/flamestore'],
      cmdclass={'install_headers': install_headers},
      headers=['swift/flamestore.h'] # swift file considered a header
    )
