namespace AuthService.Middlewares;

public sealed class ApiLoggingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<ApiLoggingMiddleware> _logger;

    public ApiLoggingMiddleware(RequestDelegate next, ILogger<ApiLoggingMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        try
        {
            await _next(context);

            if (context.Response.StatusCode is >= 200 and <= 299)
            {
                _logger.LogInformation($"Request {context.Request.Method} {context.Request.Path} succeeded with status code {context.Response.StatusCode}.");
            }
            else
            {
                _logger.LogWarning($"Request {context.Request.Method} {context.Request.Path} failed with status code {context.Response.StatusCode}.");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"An exception occurred for request {context.Request.Method} {context.Request.Path}.");
            throw;
        }
    }
}