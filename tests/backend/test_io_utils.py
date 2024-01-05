import unittest
from unittest.mock import mock_open, patch
from googly_eyes.backend.lib.utils.io_utils import load_config


class TestLoadConfig(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="key1: value1\nkey2: value2")
    @patch("os.path.join", return_value="/path/to/config/config.yml")
    def test_load_config(self, mock_os_path_join, mock_open):
        expected_config = {'key1': 'value1', 'key2': 'value2'}

        result = load_config()

        # Assert
        mock_os_path_join.assert_called_once_with("config", "config.yml")
        mock_open.assert_called_once_with("/path/to/config/config.yml", 'r')
        self.assertEqual(result, expected_config)

    @patch("os.path.join", return_value="/path/to/nonexistent/config/config.yml")
    def test_load_config_file_not_found(self, mock_os_path_join):
        with self.assertRaises(FileNotFoundError):
            load_config()

        mock_os_path_join.assert_called_once_with("config", "config.yml")


if __name__ == '__main__':
    unittest.main()
