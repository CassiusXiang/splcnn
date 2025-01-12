use serde_json::json;
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::RpcBlockConfig;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_transaction_status::{TransactionDetails, UiTransactionEncoding};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let client = RpcClient::new(
        "https://mainnet.helius-rpc.com/?api-key=79d5a1a2-bb2a-4ec0-a22e-f2d229cb66ec",
    );
    let current_slot = client
        .get_slot_with_commitment(CommitmentConfig::confirmed())
        .expect("cant slot");
    println!("now slot: {}", current_slot);

    let num_slots = 100_000;

    let starting_slot = if current_slot > num_slots {
        current_slot - num_slots + 1
    } else {
        0
    };
    println!("it slots f {} to {}", starting_slot, current_slot);
    let file = File::create("transactions.jsonl").expect("cant mkfile");
    let mut writer = BufWriter::new(file);

    for slot in starting_slot..=current_slot {
        if slot % 1000 == 0 {
            println!("processing slot: {}", slot);
        }

        match client.get_block_with_config(
            slot,
            RpcBlockConfig {
                encoding: Some(UiTransactionEncoding::Json),
                transaction_details: Some(TransactionDetails::Full),
                rewards: Some(false),
                ..Default::default()
            },
        ) {
            Ok(block) => {
                if let Some(transactions) = block.transactions {
                    for transaction_with_meta in transactions {
                        let transaction = transaction_with_meta.transaction;
                        let transaction_json = serde_json::to_string(&transaction);
                        match transaction_json {
                            Ok(json_str) => {
                                if let Err(e) = writeln!(writer, "{}", json_str) {
                                    eprintln!("cant,tx->d: {:?}", e);
                                }
                            }
                            Err(e) => {
                                eprintln!("cant序列化: {:?}", e);
                            }
                        }
                    }
                } else {
                    println!("Slot {} notx😡", slot);
                }
            }
            Err(err) => {
                println!("no slot {} of: {:?}", slot, err);
            }
        }
    }

    if let Err(e) = writer.flush() {
        eprintln!("shit: {:?}", e);
    }

    println!("txs save in transactions.json");
}
