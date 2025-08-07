using AuthNuget.Security;
using AuthService.Commands.NewUser;
using AuthService.Commands.Password;
using AuthService.Commands.Seedwork;
using AuthService.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AuthService.Controllers;

// [Authorize(Roles = RoleConstants.AdminRole)]
[ApiController]
[Route("users")]
public sealed class UsersController : ControllerBase
{
    private readonly ICommandDispatcher _commandDispatcher;

    public UsersController(ICommandDispatcher commandDispatcher)
    {
        _commandDispatcher = commandDispatcher;
    }

    [HttpPatch("password")]
    public async Task<ActionResult> AdminChangeUserPassword([FromBody] AdminChangePasswordDto adminChangePasswordDto)
    {
        var changePassword = new ChangePassword(adminChangePasswordDto.UserName, adminChangePasswordDto.OldPassword, adminChangePasswordDto.NewPassword);

        var result = await _commandDispatcher.DispatchAsync(changePassword);

        if (result.IsSuccess()) return Ok();

        return BadRequest(result.Exception!.Message);
    }

    [HttpPost]
    public async Task<ActionResult> CreateUserForRole([FromBody] CreateUser createUser)
    {
        var result = await _commandDispatcher.DispatchAsync(createUser);

        if (result.IsSuccess()) return Ok();

        return BadRequest(result.Exception!.Message);
    }
}