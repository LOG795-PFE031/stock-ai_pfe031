
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from services.data_ingestion.data_service import DataService

@pytest.fixture
def data_service():
    """Instance de DataService pour les tests."""
    service = DataService()
    service._initialized = True
    return service

@pytest.mark.asyncio
async def test_get_stock_name_success(data_service):
    """Test de récupération du nom d'une action."""
    symbol = "AAPL"
    
    # Mock de yfinance
    mock_ticker = MagicMock()
    mock_ticker.info = {"shortName": "Apple Inc."}
    
    with patch('yfinance.Ticker', return_value=mock_ticker), \
         patch('pandas.read_csv') as mock_read_csv, \
         patch('pandas.DataFrame.to_csv') as mock_to_csv:
        
        # Mock du fichier CSV existant
        mock_df = pd.DataFrame({"name": ["Apple Inc."]}, index=["AAPL"])
        mock_read_csv.return_value = mock_df
        
        result = data_service.get_stock_name(symbol)
        
        assert result == "Apple Inc."

@pytest.mark.asyncio
async def test_get_current_price_success(data_service):
    """Test de récupération du prix actuel avec succès."""
    symbol = "AAPL"
    
    # Mock des données de retour avec la structure correcte
    mock_df = pd.DataFrame({
        "Date": [datetime.now()],
        "Close": [123.45],
        "Open": [120.0],
        "High": [125.0],
        "Low": [119.0],
        "Volume": [1000000],
        "Adj Close": [123.40]
    })
    
    with patch.object(data_service, '_get_stock_data', AsyncMock(return_value=mock_df)), \
         patch.object(data_service, 'get_stock_name', return_value="Apple Inc."):
        
        result = await data_service.get_current_price(symbol)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert result[1] == "Apple Inc."

@pytest.mark.asyncio
async def test_get_recent_data_success(data_service):
    """Test de récupération des données récentes."""
    symbol = "AAPL"
    days_back = 5
    
    # Mock des données de retour
    mock_df = pd.DataFrame({
        "Date": [datetime.now() - timedelta(days=i) for i in range(5)],
        "Open": [120.0, 121.0, 122.0, 123.0, 124.0],
        "High": [125.0, 126.0, 127.0, 128.0, 129.0],
        "Low": [119.0, 120.0, 121.0, 122.0, 123.0],
        "Close": [123.45, 124.45, 125.45, 126.45, 127.45],
        "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        "Adj Close": [123.40, 124.40, 125.40, 126.40, 127.40]
    })
    
    with patch.object(data_service, '_get_stock_data', AsyncMock(return_value=mock_df)), \
         patch.object(data_service, 'get_stock_name', return_value="Apple Inc."):
        
        result = await data_service.get_recent_data(symbol, days_back)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert result[1] == "Apple Inc."
        assert len(result[0]) == 5

@pytest.mark.asyncio
async def test_get_historical_stock_prices_success(data_service):
    """Test de récupération des données historiques."""
    symbol = "AAPL"
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    # Mock des données de retour
    mock_df = pd.DataFrame({
        "Date": [datetime.now() - timedelta(days=i) for i in range(10)],
        "Open": [120.0 + i for i in range(10)],
        "High": [125.0 + i for i in range(10)],
        "Low": [119.0 + i for i in range(10)],
        "Close": [123.45 + i for i in range(10)],
        "Volume": [1000000 + i*100000 for i in range(10)],
        "Adj Close": [123.40 + i for i in range(10)]
    })
    
    with patch.object(data_service, '_get_stock_data', AsyncMock(return_value=mock_df)), \
         patch.object(data_service, 'get_stock_name', return_value="Apple Inc."):
        
        result = await data_service.get_historical_stock_prices(symbol, start_date, end_date)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert result[1] == "Apple Inc."

@pytest.mark.asyncio
async def test_initialize_success(data_service):
    """Test d'initialisation avec succès."""
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        await data_service.initialize()
        
        assert data_service._initialized is True
        mock_mkdir.assert_called_once()

@pytest.mark.asyncio
async def test_cleanup_success(data_service):
    """Test de nettoyage avec succès."""
    data_service._initialized = True
    
    await data_service.cleanup()
    
    assert data_service._initialized is False

@pytest.mark.asyncio
async def test_calculate_change_percent_success(data_service):
    """Test de calcul du pourcentage de changement."""
    symbol = "AAPL"
    
    # Mock des données de retour avec 2 jours (aujourd'hui et hier)
    # Les données sont triées par date décroissante (le plus récent en premier)
    mock_df = pd.DataFrame({
        "Date": [datetime.now(), datetime.now() - timedelta(days=1)],
        "Close": [120.0, 100.0]  # Prix actuel: 120, prix hier: 100
    })
    
    with patch.object(data_service, 'get_recent_data', AsyncMock(return_value=(mock_df, "Apple Inc."))):
        result = await data_service.calculate_change_percent(symbol)
        
        assert isinstance(result, float)
        assert result == 20.0  # (120 - 100) / 100 * 100 = 20%

@pytest.mark.asyncio
async def test_calculate_change_percent_negative_change(data_service):
    """Test de calcul du pourcentage de changement négatif."""
    symbol = "AAPL"
    
    # Mock des données avec une baisse
    mock_df = pd.DataFrame({
        "Date": [datetime.now(), datetime.now() - timedelta(days=1)],
        "Close": [80.0, 100.0]  # Prix actuel: 80, prix hier: 100
    })
    
    with patch.object(data_service, 'get_recent_data', AsyncMock(return_value=(mock_df, "Apple Inc."))):
        result = await data_service.calculate_change_percent(symbol)
        
        assert isinstance(result, float)
        assert result == -20.0  # (80 - 100) / 100 * 100 = -20%

@pytest.mark.asyncio
async def test_calculate_change_percent_no_data(data_service):
    """Test de calcul du pourcentage de changement sans données."""
    symbol = "AAPL"
    
    # Mock d'un DataFrame vide
    mock_df = pd.DataFrame()
    
    with patch.object(data_service, 'get_recent_data', AsyncMock(return_value=(mock_df, "Apple Inc."))):
        result = await data_service.calculate_change_percent(symbol)
        
        assert result is None

@pytest.mark.asyncio
async def test_calculate_change_percent_insufficient_data(data_service):
    """Test de calcul du pourcentage de changement avec données insuffisantes."""
    symbol = "AAPL"
    
    # Mock avec seulement 1 jour de données
    mock_df = pd.DataFrame({
        "Date": [datetime.now()],
        "Close": [120.0]
    })
    
    with patch.object(data_service, 'get_recent_data', AsyncMock(return_value=(mock_df, "Apple Inc."))):
        result = await data_service.calculate_change_percent(symbol)
        
        assert result is None

@pytest.mark.asyncio
async def test_get_nasdaq_stocks_success(data_service):
    """Test de récupération des actions NASDAQ."""
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.text = """
        Symbol,Name,Last Sale,Net Change,% Change,Market Cap,Country,IPO Year,Volume,Sector,Industry
        AAPL,Apple Inc.,150.00,2.50,1.69%,2500000000000,USA,1980,50000000,Technology,Consumer Electronics
        MSFT,Microsoft Corporation,300.00,5.00,1.67%,2000000000000,USA,1986,30000000,Technology,Software
        """
        mock_get.return_value = mock_response
        
        result = await data_service.get_nasdaq_stocks()
        
        assert isinstance(result, dict)
        assert "count" in result
        assert "data" in result

@pytest.mark.asyncio
async def test_health_check_not_initialized(data_service):
    """Test du health check quand le service n'est pas initialisé."""
    data_service._initialized = False
    
    result = await data_service.health_check()
    
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "unhealthy"
    assert "Service not initialized" in result["message"]

@pytest.mark.asyncio
async def test_get_stock_name_from_cache(data_service):
    """Test de récupération du nom d'action depuis le cache."""
    symbol = "AAPL"
    
    # Mock du fichier CSV existant avec le symbole
    mock_df = pd.DataFrame({"name": ["Apple Inc."]}, index=["AAPL"])
    
    with patch('pandas.read_csv', return_value=mock_df), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv:
        
        result = data_service.get_stock_name(symbol)
        
        assert result == "Apple Inc."
        # Vérifier que to_csv n'a pas été appelé (pas de mise à jour du cache)
        mock_to_csv.assert_not_called()

@pytest.mark.asyncio
async def test_get_recent_data_with_default_days(data_service):
    """Test de récupération des données récentes avec les jours par défaut."""
    symbol = "AAPL"
    
    # Mock des données de retour
    mock_df = pd.DataFrame({
        "Date": [datetime.now() - timedelta(days=i) for i in range(10)],
        "Open": [120.0 + i for i in range(10)],
        "High": [125.0 + i for i in range(10)],
        "Low": [119.0 + i for i in range(10)],
        "Close": [123.45 + i for i in range(10)],
        "Volume": [1000000 + i*100000 for i in range(10)],
        "Adj Close": [123.40 + i for i in range(10)]
    })
    
    with patch.object(data_service, '_get_stock_data', AsyncMock(return_value=mock_df)), \
         patch.object(data_service, 'get_stock_name', return_value="Apple Inc."):
        
        result = await data_service.get_recent_data(symbol)  # Sans spécifier days_back
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert result[1] == "Apple Inc."
