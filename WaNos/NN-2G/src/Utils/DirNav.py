"""

This Module has Useful Functions For the management of directories. Intended to
make the directory management apply for different systems.

All functions are implemented using the path str by methods in os.path. Use
pathlib is depreciated since it is currently unable to automatically convert to
path str, which is used in ase package.

"""
import os
import shutil




def get_parent_dir(path):
    parent_path_str = os.path.join(path, '..')
    parent_path_abs = os.path.abspath(parent_path_str)
    return parent_path_abs

def get_project_dir():
    file_path = os.path.dirname(os.path.realpath(__file__))
    return get_parent_dir(file_path)

def get_namespace_dir():
    pj_dir = get_project_dir()
    ns_dir      = get_parent_dir(pj_dir)
    return ns_dir


def get_model_dir(model_name):
    ns_dir = get_namespace_dir()
    model_dir = os.path.join(ns_dir, 'models', model_name)
    return model_dir

def mkdir_model(model_name):
    """Create a directory for containing the training set for the models.

            Args:
                model_name: name of the model to be used.

            Returns:
                None.
                Creates a folder in the default model folder (/model/)
                /model/model_name

            Raises:
                OSError: If the directory already exits.

    """
    project_dir = get_project_dir()

    model_path = os.path.join(project_dir, 'model', model_name)

    try:
        os.mkdir(model_path)
    except:
        raise OSError("Not able to create directory:", model_path,". It may " +
                        "already exits")

    print("Successfully creates directory ", model_path)

    return


def rmdir_model(model_name):
    """See mkdir_model docstring.

    """
    project_dir = get_project_dir()
    model_path = os.path.join(project_dir, 'model', model_name)

    try:
        shutil.rmtree(model_path)
    except:
        raise OSError("NOt able to delete the directory:", model_path)

    print("Successfully deletes the directory ", model_path)

    return
