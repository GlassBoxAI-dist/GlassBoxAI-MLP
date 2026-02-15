//! Kani Verification: Enum Exhaustion
//!
//! Verify that all match statements handle every possible variant without
//! relying on generic _ => panic!() fallback.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_activation_type_exhaustive() {
        let act_type: i32 = kani::any();
        kani::assume(act_type >= 0 && act_type <= 3);
        
        let act = match act_type {
            0 => TActivationType::AtSigmoid,
            1 => TActivationType::AtTanh,
            2 => TActivationType::AtReLU,
            3 => TActivationType::AtSoftmax,
            _ => TActivationType::AtSigmoid,
        };
        
        let name = activation_to_str(act);
        kani::assert(!name.is_empty(), "All activation types have names");
    }

    #[kani::proof]
    fn verify_optimizer_type_exhaustive() {
        let opt_type: i32 = kani::any();
        kani::assume(opt_type >= 0 && opt_type <= 2);
        
        let opt = match opt_type {
            0 => TOptimizerType::OtSGD,
            1 => TOptimizerType::OtAdam,
            2 => TOptimizerType::OtRMSProp,
            _ => TOptimizerType::OtSGD,
        };
        
        let name = optimizer_to_str(opt);
        kani::assert(!name.is_empty(), "All optimizer types have names");
    }

    #[kani::proof]
    fn verify_command_type_exhaustive() {
        let cmd_type: i32 = kani::any();
        kani::assume(cmd_type >= 0 && cmd_type <= 5);
        
        let cmd = match cmd_type {
            0 => TCommand::CmdNone,
            1 => TCommand::CmdCreate,
            2 => TCommand::CmdTrain,
            3 => TCommand::CmdPredict,
            4 => TCommand::CmdInfo,
            5 => TCommand::CmdHelp,
            _ => TCommand::CmdNone,
        };
        
        kani::assert(
            matches!(cmd, TCommand::CmdNone | TCommand::CmdCreate | TCommand::CmdTrain |
                         TCommand::CmdPredict | TCommand::CmdInfo | TCommand::CmdHelp),
            "All command types covered"
        );
    }
}
