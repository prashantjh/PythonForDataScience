# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:36:16 2018

@author: jhapr

MODULE - Highest level program organization unit, 
    which packages program code and data for reuse, 
    and provides self-contained namespaces that minimize variable clashes across your programs

"""

# =============================================================================
# MODULES 
# 
# - Each file is a module, and modules import other modules to use the names they define
# - Modules are processed with two statements and one important function:
#     import - Lets a client (importer) fetch a module as a whole
#     from - Allows client to fetch particular names from a module
#     imp.reload - A way to reload a module's code without stopping Python
#     
# - Modules provide an easy way to organize components into a system by serving a self-contained packages of variables known as namespaces
# - All the names defined at the top level of a module file become attributes of the imported module object
# 
# 
# - The Modules have at least three roles:
#     Code reuse
#     System namespace partioning
#     Implementing shared services or data
# 
# - Module namespaces can be accessed via the attribute __dict__ or dir(M)
# 
# =============================================================================


# =============================================================================
# How Import Works:
# 
# - Import is a runtime operation that performs 3 distinct steps:
#     1. Find the module file
#     2. Compile it to byte code (if needed)
#     3. Run the module's code to build the object it defines
#     
# - Python uses a standard module search parth and know file types to locate the module
#
# - Python only imports once in a program and ignores further import
#    To re-import/load a module/package use:
#       > from imp import reload
#       > reload(module_name)
# =============================================================================


# =============================================================================
# Package __init__.py files
# 
# -For a directory structure, dir0\dir1\dir2\mod.py and import statement
#     import dir1.dir2.mod
#   -the following rules apply:
#     - dir1 and dir2 both must contain an __init__.py file.
#     - dir0, the container, does not require an __init__.py file; this file will simply be
#     ignored if present.
#     - dir0, not dir0\dir1, must be listed on the module search path sys.path
# 
# - The __init__.py file serves as a hook for:
#     - package initialization-time actions
#     - declares a directory as a Python package
#     - Generates a module namespace for a directory
#     - Implements the behavior of the from *  (i.e. from .. import *)
# 
# - The first time a Python program imports through a directory, 
#     it automatically runs all the code in the direcotory's __init__.py file
#     These files are a natural place to put code to initialize the state required by files in a package
#     
# - Package __init__.py files are also partly present to declare that a directory is a Python package
# 
# - Don't confuse package __init__.py with the class __init_ constructor
# =============================================================================



# =============================================================================
# Package imports: (Oldest to Newest)
# 
#     Basic module imports: import mod, from mod import attr
#     Package imports: import dir1.dir2.mod, from dir1.mod import attr
#     Package-relative imports: from . import mod (relative), import mod (absolute)
#     Namespace packages: import splitdir.mod
# 
# - Namespace package model allows packages to span multiple directories,
#     and requires no initialization file
# =============================================================================




# =============================================================================
# Module Design Concepts:
#     
# - You're always in a module in Python
#     Even code typed at the interactive prompt goes in an built-in module, __main__
#     
# - Minimize module coupling: global variables
#     Modules should be independent of global variables used within other modules, 
#     except for functions and classes imported from them
# 
# - Maximize module cohesion: unified purpose
#     If all components of a module share a general purpose, you're less likely to depend on external names
#     
# - Modules should rarely change other module's variables
# =============================================================================



# =============================================================================
# Data Hiding in Modules:
# 
# - Minimize 'from *' Damage: _X and __all__
#     - prefixing names with underscore (_X) prevents them from being copied out when a client imports a module's name with a from * statement
#     - Alternatively, by assigning a list of variable name strings to the variable __all__ at the top level of the module, you can list down what should be copied when from * statement is used
#     - __all__ identifies names to be copied, while _X identifies names NOT to be copied
#     
# 
# Enabling future language features: __future__
# 
# - Changes to the language that may potentially break existing code are usually introduced gradually in Python
# - Use from __future__ import featurename to turn on optional extensions, which are disabled by default
# =============================================================================



# =============================================================================
# Mixed usage modes: __name__ and __main__
# 
# - These helps import a file as a module and run it as a standalone program
# - Each module has a built-in attribute called __name__, which Python creates and assigns automatically as follows:
#     - If the file is being run as a top-level program file, __name__ is set to the string
#         "__main__" when it starts.
#     - If the file is being imported instead, __name__ is set to the module’s name as known by its clients
# - The upshot is that a module can test its own __name__ to determine whether it's being run or imported
#     Eg: runme.py
#         def tester():
#             print("It's Hello World Program")
#         
#         if __name__ == '__main__':
#             tester()
#             
#         Above is executed only when run not when imported
#     Execution example:
#         import runme
#         runme.tester()
#         -- tester function is executed because of import
#         
#         python runme.py
#         -- __name__ variables helps execute tester()
#     
# =============================================================================













