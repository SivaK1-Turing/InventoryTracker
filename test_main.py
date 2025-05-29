# test_main.py
from typer.testing import CliRunner
from inventorytracker.main import app

runner = CliRunner()

def test_app():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Inventory Tracker" in result.stdout