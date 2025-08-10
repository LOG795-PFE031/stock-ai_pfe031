namespace AuthService.Dtos;

public sealed record PasswordChangeDto(string NewPassword, string OldPassword);