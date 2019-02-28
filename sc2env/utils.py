import tensorflow as tf


def save_graph(path, sess=None):
    sess = sess or tf.get_default_session()
    writer = tf.summary.FileWriter(path, sess.graph)
    writer.flush()

class Arguments:
    def __init__(self, **kwargs):
        for name in kwargs:
            if isinstance(kwargs[name], dict):
                setattr(self, name, Arguments(**kwargs[name]))
            else:
                setattr(self, name, kwargs[name])

    def __eq__(self, other):
        if not isinstance(other, Arguments):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__

    def pop(self, name):
        v = getattr(self, name)
        self.__delattr__(name)
        return v

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))

    def __iter__(self):
        for attr in self.__dict__:
            yield attr, self.__dict__[attr]

    def _get_kwargs(self):
        return sorted(self.__dict__.items())

    def _get_args(self):
        return []

    def update(self, kv):
        self.__dict__.update(kv)