using AuthNuget.Security;
using AuthService.Dtos;
using AuthService.Queries.Jwt;
using AuthService.Queries.Seedwork;
using Microsoft.AspNetCore.Mvc;

namespace AuthService.Controllers;

[ApiController]
[Route("auth")]
public class AuthController : ControllerBase
{
    private readonly IQueryDispatcher _queryDispatcher;
    private readonly ILogger<AuthController> _logger;

    public AuthController(IQueryDispatcher queryDispatcher, ILogger<AuthController> logger)
    {
        _queryDispatcher = queryDispatcher;
        _logger = logger;
    }

    [HttpGet("publickey")]
    public ActionResult GetPublicKey()
    {
        return Ok(new ServerPublicKey (){ PublicKey = RsaKeyStorage.Instance.PublicKey });
    }

    [HttpPost("signin")]
    public async Task<ActionResult> Login([FromBody] UserCredentials userCredentials)
    {
        var createJwt = new GetJwtForCredentials(userCredentials.Username, userCredentials.Password);

        var result = await _queryDispatcher.DispatchAsync<GetJwtForCredentials, string>(createJwt, CancellationToken.None);

        if (result.IsSuccess()) return Ok(result.Content);

        _logger.LogError(result.Exception!.Message);

        return Forbid();
    }
}