using AuthService.Commands.Seedwork;
using AuthService.Domain;
using AuthService.Monads;
using Microsoft.AspNetCore.Identity;

namespace AuthService.Commands.Password;

public sealed class ChangePasswordHandler : ICommandHandler<ChangePassword>
{
    private readonly UserManager<UserPrincipal> _userManager;
    private readonly ILogger<ChangePasswordHandler> _logger;

    public ChangePasswordHandler(UserManager<UserPrincipal> userManager, ILogger<ChangePasswordHandler> logger)
    {
        _userManager = userManager;
        _logger = logger;
    }
    public async Task<Result> Handle(ChangePassword command, CancellationToken cancellation)
    {
        var user = await _userManager.FindByNameAsync(command.Username);

        if (user is null)
        {
            return Result.Failure("Invalid Username");
        }

        var passwordCheck = await _userManager.CheckPasswordAsync(user, command.OldPassword);

        if (!passwordCheck)
        {
            return Result.Failure("Invalid Password");
        }

        var passwordChangeResult = await _userManager.ChangePasswordAsync(user, command.OldPassword, command.NewPassword);

        if (!passwordChangeResult.Succeeded)
        {
            return Result.Failure("Password Change Failed");
        }

        _logger.LogInformation("Password changed for user {Username}", user.UserName);

        await _userManager.UpdateAsync(user);

        return Result.Success();
    }
}