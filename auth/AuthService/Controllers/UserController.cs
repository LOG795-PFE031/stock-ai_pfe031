using AuthNuget.Security;
using AuthService.Commands.Password;
using AuthService.Commands.Seedwork;
using AuthService.Dtos;
using AuthService.Extensions;
using AuthService.Queries.Seedwork;
using AuthService.Queries.User;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AuthService.Controllers;

[Authorize(Roles = $"{RoleConstants.Client}, {RoleConstants.AdminRole}")]
[ApiController]
[Route("user")]
public sealed class UserController : ControllerBase
{
    [HttpPatch("password")]
    public async Task<ActionResult> ChangePassword([FromBody] PasswordChangeDto passwordChangeDto, ICommandDispatcher commandDispatcher)
    {
        var changePassword = new ChangePassword(User.GetUsername(), passwordChangeDto.OldPassword, passwordChangeDto.NewPassword);

        var result = await commandDispatcher.DispatchAsync(changePassword);

        if (result.IsSuccess()) return Ok();

        return BadRequest(result.Exception!.Message);
    }

    [HttpGet("wallet")]
    public async Task<ActionResult<WalletId>> Get(IQueryDispatcher queryDispatcher)
    {
        var result = await queryDispatcher.DispatchAsync<GetUserWalletId, string>(new GetUserWalletId(User.GetUsername()));

        if (result.IsSuccess()) return Ok(new WalletId(result.Content!));

        return BadRequest(result.Exception!.Message);
    }

    [HttpGet("validate")]
    public ActionResult Validate()
    {
        return Ok();
    }

    public sealed record WalletId(string Value);
}